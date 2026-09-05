"""Cached-output falsification diagnostic, not an input ablation or test score."""
import numpy as np
import torch
from mtare_topo.representation.gse_point_axis_loss import axis_set_metrics
from mtare_topo.evaluation.gse_axis_error_decomposition import decompose_axes


def same_parent_derangement(manifest):
    """Fixed half-cycle shift within each18-row parent, no outcome selection."""
    if len(manifest)!=180:raise ValueError("exact180 observations required")
    keys=[(r["task"],r["row_index"]) for r in manifest]
    if len(set(keys))!=180:raise ValueError("duplicate observation")
    parents=sorted({r["task"] for r in manifest})
    if len(parents)!=10:raise ValueError("exact10 parents required")
    mapping=np.empty(180,dtype=np.int64)
    for parent in parents:
        indices=[i for i,r in enumerate(manifest) if r["task"]==parent]
        if len(indices)!=18:raise ValueError("exact18 rows per parent required")
        # Manifest order is frozen; no teachers, predictions, IDs or scores used.
        for j,index in enumerate(indices):mapping[index]=indices[(j+9)%18]
    return mapping.tolist()


def query_redundancy(axes):
    value=np.asarray(axes)
    if value.shape!=(32,3,3) or not np.isfinite(value).all():raise ValueError("finite32 axes required")
    difference=np.abs(value[:,None].astype(np.float64)-value[None].astype(np.float64)).mean(axis=(-1,-2))
    reverse=np.abs(value[:,None].astype(np.float64)-value[None,:,::-1].astype(np.float64)).mean(axis=(-1,-2))
    cost=np.minimum(difference,reverse);np.fill_diagonal(cost,np.inf)
    nearest=cost.min(axis=1)
    return {"nearest_other_query_coordinate_mae_m":nearest.tolist(),
            "exact_duplicate_unordered_pairs":int(np.count_nonzero(np.triu(cost==0,k=1)))}


def score_observation(axes,target,mask):
    value=np.asarray(axes)
    fit=axis_set_metrics(torch.from_numpy(value[None]),target,mask)[0]
    layout=[]
    for match in fit["matching"]:
        teacher=target[0,match["target_index"]].numpy()
        if match["reversed"]:teacher=teacher[::-1].copy()
        layout.append({"prediction_index":match["prediction_index"],"target_index":match["target_index"],
            **decompose_axes(value[match["prediction_index"]],teacher)})
    fields=("transverse_rms_m","undirected_direction_error_deg","polyline_symmetric_m")
    means={};counts={}
    for field in fields:
        valid=[r[field] for r in layout if r[field] is not None]
        means[field]=float(np.mean(valid)) if valid else None
        counts[field]={"resolved":len(valid),"unresolved":len(layout)-len(valid)}
    return {"fit":fit,"layout":layout,"layout_mean":means,"layout_counts":counts,
            "redundancy":query_redundancy(value)}


def summarize(rows,manifest):
    if len(rows)!=len(manifest) or not rows:raise ValueError("row count mismatch")
    def aggregate(indices):
        selected=[rows[i] for i in indices]
        layout={}
        for field in selected[0]["layout_mean"]:
            values=[r["layout_mean"][field] for r in selected if r["layout_mean"][field] is not None]
            layout[field]={"mean":float(np.mean(values)) if values else None,"observations_resolved":len(values),
                "fragments_resolved":sum(r["layout_counts"][field]["resolved"] for r in selected),
                "fragments_unresolved":sum(r["layout_counts"][field]["unresolved"] for r in selected)}
        counts=np.zeros(32,dtype=np.int64)
        for row in selected:
            for match in row["fit"]["matching"]:counts[match["prediction_index"]]+=1
        total=int(counts.sum());prob=counts/total
        return {"observations":len(selected),"coordinate_mae_m":float(np.mean([r["fit"]["coordinate_mae_m"] for r in selected])),
            "point_mean_euclidean_m":float(np.mean([r["fit"]["point_mean_euclidean_m"] for r in selected])),
            "layout":layout,"query_match_counts":counts.tolist(),"matched_targets":total,
            "surplus_queries":sum(r["fit"]["unmatched_predictions"] for r in selected),
            "query_use_effective_count":float(1/np.square(prob).sum()),
            "nearest_other_query_coordinate_mae_m":float(np.mean([r["redundancy"]["nearest_other_query_coordinate_mae_m"] for r in selected])),
            "exact_duplicate_unordered_pairs":sum(r["redundancy"]["exact_duplicate_unordered_pairs"] for r in selected)}
    parents={parent:aggregate([i for i,r in enumerate(manifest) if r["task"]==parent]) for parent in sorted({r["task"] for r in manifest})}
    return {"macro":aggregate(range(len(rows))),"parents":parents}


def compare(correct,shuffled):
    if correct["parents"].keys()!=shuffled["parents"].keys():raise ValueError("parent mismatch")
    deltas={p:shuffled["parents"][p]["coordinate_mae_m"]-correct["parents"][p]["coordinate_mae_m"] for p in correct["parents"]}
    return {"coordinate_shuffled_minus_correct_m":shuffled["macro"]["coordinate_mae_m"]-correct["macro"]["coordinate_mae_m"],
        "per_parent_coordinate_shuffled_minus_correct_m":deltas,
        "parents_correct_better":sum(v>0 for v in deltas.values()),
        "parents_tied":sum(v==0 for v in deltas.values()),
        "parents_correct_worse":sum(v<0 for v in deltas.values()),
        "scientific_gate_pass":False,"interpretation":"FIT output-correspondence diagnostic only; does not separate sensor geometry from pose/route correlations or memorization. Surplus queries are not false-positive scored."}
